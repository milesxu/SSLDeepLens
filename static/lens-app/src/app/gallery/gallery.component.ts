import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-gallery',
  templateUrl: './gallery.component.html',
  styleUrls: ['./gallery.component.css']
})
export class GalleryComponent implements OnInit {
  base_name = 'assets/combine123/';
  base_num = 100000;
  offset = 9000;
  length = 1024;
  batch_len = 128;
  batch_num = 8;
  images: string[] = [];
  batched_images: string[] = [];
  index = 1;
  constructor() {}

  ngOnInit() {
    const start = this.base_num + this.offset;
    for (let i = 0; i < this.length; ++i) {
      const image_name = `${this.base_name}ground_based_${start + i}.png`;
      this.images.push(image_name);
    }
    this.batch_num = this.length / this.batch_len;
    this.batched_images = this.images.slice(0, this.batch_len);
  }

  changePageIndex(event: number): void {
    const start = (event - 1) * this.batch_len;
    this.batched_images = this.images.slice(start, start + this.batch_len);
  }
}
